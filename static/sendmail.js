function sendEmail(){
    Email.send({
    Host : "smtp.elasticemail.com",
    Username : "shanigarapuvinay09@gmail.com",
    Password : "5489DCAF63BFB55B6D2DB3890C40C159B296",
    To : 'climatechangeanalysisvbb@gmail.com',
    From : "shanigarapuvinay09@gmail.com",
    Subject : "This is the subject",
    Body : "And this is the body"
}).then(
  message => alert(message)
);
}